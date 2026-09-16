#!/usr/bin/env python3
"""GRPO interactivity alignment for Moshi, with LoRA adapters and an LLM judge.

Reproduces the Kyutai Multi-Faceted Interactivity Alignment paper
(arxiv 2606.11167) using segment-based training:

  1. Load per-axis segment datasets (D_pause, D_turn, D_bc, D_int)
  2. Each epoch: sample segments, optionally prepend context audio
  3. Feed user-side audio to Moshi -> generate G rollouts in ONE batched pass
  4. VAD the generated audio -> deterministic axis reward
  5. Reshape each rollout into a Full-Duplex-Bench-JA row -> LLM judge reward
  6. Per-axis reward decoupling -> GRPO policy update on the LoRA adapters

Two GPUs: this process trains on GPU 0, and a resident vLLM server holds the
27B judge on GPU 1 (see scripts/2026-09-04/grpo_judge_lora.pbs). The judge is
reached over HTTP, so its weights never enter this process's allocator.

    python scripts/grpo/train_grpo.py --config configs/grpo_judge_lora.yaml
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
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.grpo.config import GRPOConfig
from scripts.grpo.judge_prompt import FLAG_KEYS, SCORE_KEYS
from scripts.grpo.llm_judge import judge_from_config
from scripts.grpo.lora import (
    adapters_disabled,
    apply_lora,
    freeze_base,
    load_lora_checkpoint,
    lora_parameters,
    save_lora_checkpoint,
)
from scripts.grpo.rewards import (
    decoupling_normalize,
    judge_reward_axes,
    reward_backchannel_f1,
    reward_pause_handling,
    reward_turn_taking,
    reward_user_interruption,
)
from scripts.grpo.rollout_to_judge_input import build_packed_row

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

AXES = ["pause", "turn_taking", "backchannel", "interruption"]
SAMPLE_RATE = 24_000

# Submodule name fragments for the temporal transformer's text output
# projection. The logits it produces are the only thing GRPO differentiates, so
# if none of these match, training cannot proceed and must say so.
TEXT_HEAD_HINTS = ("text_linear", "text_out", "text_head", "output_text", "text_proj")


# -- Seed -------------------------------------------------------------

def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# -- Segment dataset loading -----------------------------------------

def load_segment_datasets(segment_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load per-axis segment JSONL files produced by segment_extractor.py."""
    seg_dir = Path(segment_dir)
    datasets: dict[str, list[dict[str, Any]]] = {}
    for axis in AXES:
        path = seg_dir / f"{axis}_segments.jsonl"
        items: list[dict[str, Any]] = []
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        datasets[axis] = items
    return datasets


def load_segment_audio(
    segment: dict[str, Any],
    context_sec: float,
    context_prob: float,
    rng: random.Random,
) -> tuple[np.ndarray, float]:
    """Load user-side audio for a segment, optionally with a context prefix.

    Returns (user_pcm, segment_offset_sec), where segment_offset_sec is where
    the segment proper starts inside the returned audio (= context length).
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
        ctx_samples = int(ctx_len_sec * sr)
        start_sample = max(0, seg_start_sample - ctx_samples)
        ctx_len_sec = (seg_start_sample - start_sample) / sr
    else:
        start_sample = seg_start_sample

    return user_channel[start_sample:seg_end_sample].copy(), ctx_len_sec


# -- Model loading ----------------------------------------------------

def _filtered_kwargs(fn: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs the callable actually accepts.

    The moshi loader's signature has changed across releases; passing an
    unknown keyword would abort a job that is otherwise fine.
    """
    try:
        supported = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in candidates.items() if k in supported and v is not None}


def load_moshi_for_grpo(cfg: GRPOConfig, device: torch.device):
    """Load Moshi. ``moshi_weight`` is the merged SFT model when set."""
    from moshi.models import loaders

    kwargs = _filtered_kwargs(
        loaders.CheckpointInfo.from_hf_repo,
        {
            "moshi_weights": cfg.moshi_weight,
            "config_path": cfg.moshi_config,
            "lm_config": cfg.moshi_config,
        },
    )
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(cfg.hf_repo, **kwargs)
    mimi = checkpoint_info.get_mimi(device=device)
    mimi.eval()
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    dtype = torch.bfloat16 if cfg.half else torch.float32
    lm_model = checkpoint_info.get_moshi(device=device, dtype=dtype)
    return mimi, text_tokenizer, lm_model, checkpoint_info


class TextLogitsCapture:
    """Grabs the text head's output on every forward.

    GRPO differentiates the log-probability of the sampled text token, and
    moshi's streaming generator does not return logits. The previous version of
    this file read ``lm_gen.last_text_logits``, an attribute that exists in no
    release of moshi -- the ``hasattr`` guard around it meant every rollout
    collected an empty log-prob list, every GRPO loss evaluated to exactly
    zero, and training ran to completion having learned nothing. Hooking the
    module directly is what makes the loss real, so a missing head is a hard
    error here rather than a silently empty list.
    """

    def __init__(self, lm_model: nn.Module) -> None:
        self.logits: torch.Tensor | None = None
        module = self._find_text_head(lm_model)
        self.module_name, module = module
        self._handle = module.register_forward_hook(self._hook)

    @staticmethod
    def _find_text_head(lm_model: nn.Module) -> tuple[str, nn.Module]:
        matches = [
            (name, module)
            for name, module in lm_model.named_modules()
            if isinstance(module, nn.Linear)
            and any(hint in name.lower() for hint in TEXT_HEAD_HINTS)
        ]
        if not matches:
            linears = sorted(
                name
                for name, module in lm_model.named_modules()
                if isinstance(module, nn.Linear)
            )
            raise RuntimeError(
                "Could not find the text output projection. Tried name hints "
                f"{TEXT_HEAD_HINTS}. Linear modules present:\n  "
                + "\n  ".join(linears[:60])
            )
        # With several candidates the text head is the one projecting to the
        # text vocabulary, i.e. the widest output.
        return max(matches, key=lambda item: item[1].out_features)

    def _hook(self, _module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        self.logits = output

    def take(self) -> torch.Tensor:
        if self.logits is None:
            raise RuntimeError("Text head produced no logits for this step.")
        logits = self.logits
        self.logits = None
        return logits

    def close(self) -> None:
        self._handle.remove()


def _ensure_lm_gen(lm_model: nn.Module, checkpoint_info: Any, cfg: GRPOConfig):
    """Wrap lm_model in LMGen for streaming inference."""
    if hasattr(lm_model, "step") and hasattr(lm_model, "streaming"):
        return lm_model
    try:
        from moshi.models.lm import LMGen
    except ImportError:
        from moshi.models.lm_gen import LMGen

    lm_gen_cfg = getattr(checkpoint_info, "lm_gen_config", None)
    supported = set(inspect.signature(LMGen).parameters)
    kwargs: dict[str, Any] = {}
    for name in ("temp", "temp_text", "top_k", "top_k_text", "cfg_coef"):
        value = None
        if isinstance(lm_gen_cfg, dict):
            value = lm_gen_cfg.get(name)
        elif lm_gen_cfg is not None:
            value = getattr(lm_gen_cfg, name, None)
        if value is not None and name in supported:
            kwargs[name] = value
    for name, value in (
        ("temp", cfg.temp_audio),
        ("temp_text", cfg.temp_text),
        ("top_k", cfg.top_k_audio),
    ):
        if name in supported:
            kwargs[name] = value
    return LMGen(lm_model, **kwargs)


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


# -- Rollout generation -----------------------------------------------

def encode_user_audio(
    user_pcm: np.ndarray,
    mimi: Any,
    device: torch.device,
) -> list[torch.Tensor]:
    """Stream the user channel through mimi once, returning per-frame codes.

    Encoding is causal and does not depend on what Moshi generates, so it is
    done once for the whole group instead of G times. That also keeps mimi's
    encoder state at batch 1 while the decoder later runs at batch G, which a
    single shared streaming context could not do.
    """
    sample_rate = int(mimi.sample_rate)
    frame_rate = float(mimi.frame_rate)
    frame_size = int(sample_rate / frame_rate)
    n_steps = (len(user_pcm) + frame_size - 1) // frame_size

    frames: list[torch.Tensor] = []
    with torch.no_grad(), mimi.streaming(1):
        for step in range(n_steps):
            start = step * frame_size
            chunk_np = user_pcm[start: start + frame_size]
            if len(chunk_np) < frame_size:
                chunk_np = np.pad(chunk_np, (0, frame_size - len(chunk_np)))
            chunk = (
                torch.from_numpy(chunk_np).float().to(device).unsqueeze(0).unsqueeze(0)
            )
            codes = mimi.encode(chunk)
            if codes is not None:
                frames.append(codes)
    return frames


def generate_group(
    user_codes: list[torch.Tensor],
    group_size: int,
    lm_gen: Any,
    mimi: Any,
    text_tokenizer: Any,
    capture: TextLogitsCapture,
    device: torch.device,
    seed: int,
    context_offset_sec: float,
) -> list[dict[str, Any]]:
    """Generate G rollouts for one segment in a single batched streaming pass.

    The G rollouts share the same user audio, so they differ only in sampling.
    Running them as a batch instead of a loop is the difference between one
    forward per frame and G of them -- with ~500 frames per segment and G=8
    that loop was the run's dominant cost, not the judge.
    """
    frame_rate = float(mimi.frame_rate)
    text_events: list[list[dict[str, Any]]] = [[] for _ in range(group_size)]
    text_token_ids: list[list[int]] = [[] for _ in range(group_size)]
    text_log_probs: list[list[float]] = [[] for _ in range(group_size)]
    out_codes: list[torch.Tensor] = []

    _seed_all(seed)
    with torch.no_grad(), lm_gen.streaming(group_size):
        for step, codes in enumerate(user_codes):
            batch = codes.to(device).expand(group_size, -1, -1)
            out = lm_gen.step(batch)
            if out is None:
                continue
            out_codes.append(out)

            logits = capture.take()          # [B, ..., V]
            log_probs = torch.log_softmax(logits.float().reshape(group_size, -1), dim=-1)
            ids = out[:, 0, 0].tolist()
            for b, text_id in enumerate(ids):
                text_id = int(text_id)
                text_token_ids[b].append(text_id)
                text_log_probs[b].append(float(log_probs[b, text_id]))
                piece = _decode_text_piece(text_tokenizer, text_id)
                if piece is not None:
                    text_events[b].append({
                        "step": step,
                        "time_sec": round(step / frame_rate, 4),
                        "piece": piece,
                    })

    pcm = _decode_audio_codes(out_codes, mimi, group_size, device)
    n_steps = len(user_codes)
    return [
        {
            "text_events": text_events[b],
            "text_token_ids": text_token_ids[b],
            "text_log_probs": text_log_probs[b],
            "generated_pcm": pcm[b],
            "context_offset_sec": context_offset_sec,
            "total_duration_sec": n_steps / frame_rate,
            "transcript": "".join(e["piece"] for e in text_events[b]).strip(),
        }
        for b in range(group_size)
    ]


def _decode_audio_codes(
    out_codes: list[torch.Tensor],
    mimi: Any,
    group_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    """Decode each rollout's generated audio tokens back to PCM for VAD."""
    if not out_codes:
        return [np.zeros(0, dtype=np.float32) for _ in range(group_size)]

    chunks: list[np.ndarray] = []
    with torch.no_grad(), mimi.streaming(group_size):
        for codes in out_codes:
            audio_tokens = codes[:, 1:, :].to(device)  # index 0 is the text token
            try:
                pcm = mimi.decode(audio_tokens)
            except Exception:
                continue
            if pcm is not None:
                chunks.append(pcm.float().cpu().reshape(group_size, -1).numpy())

    if not chunks:
        return [np.zeros(0, dtype=np.float32) for _ in range(group_size)]
    joined = np.concatenate(chunks, axis=1)
    return [joined[b] for b in range(group_size)]


def _vad_on_generated(generated_pcm: np.ndarray, sr: int = SAMPLE_RATE):
    """Run Silero VAD on Moshi's generated audio output."""
    if len(generated_pcm) < sr * 0.1:
        return []
    from scripts.grpo.segment_extractor import _run_vad

    return [(u["start"], u["end"]) for u in _run_vad(generated_pcm, sr)]


# -- Deterministic axis reward ----------------------------------------

def compute_segment_reward(
    rollout: dict[str, Any],
    segment: dict[str, Any],
    vad_segments: list[tuple[float, float]],
    cfg: GRPOConfig,
) -> float:
    """The axis-specific deterministic reward (Kyutai axes 1-4)."""
    axis = segment["axis"]
    ctx_offset = rollout["context_offset_sec"]

    def shift(value: float) -> float:
        return value - segment["segment_start_sec"] + ctx_offset

    if axis == "pause":
        pauses = segment.get("internal_pauses", [])
        if not pauses:
            return 0.0
        total = 0.0
        for pause in pauses:
            total += reward_pause_handling(
                vad_segments, shift(pause["start"]), shift(pause["end"]),
                cfg.pause_threshold_sec,
            )
        return total / len(pauses)

    if axis == "turn_taking":
        return reward_turn_taking(vad_segments, shift(segment["user_turn_end_sec"]))

    if axis == "backchannel":
        gt_times = [shift(t) for t in segment.get("gt_backchannel_times", [])]
        return reward_backchannel_f1(
            rollout["text_events"], gt_times,
            cfg.backchannel_window_sec, cfg.backchannel_max_dur_sec,
        )

    if axis == "interruption":
        return reward_user_interruption(
            vad_segments, shift(segment["interruption_start_sec"]),
        )

    return 0.0


# -- Differentiable re-run --------------------------------------------

def recompute_log_probs(
    rollouts: list[dict[str, Any]],
    user_codes: list[torch.Tensor],
    lm_gen: Any,
    capture: TextLogitsCapture,
    device: torch.device,
    seed: int,
    context_steps: int,
) -> tuple[list[list[torch.Tensor]], float]:
    """Re-run the group with gradients on, returning differentiable log probs.

    Moshi's generator samples internally and offers no way to force a token, so
    the pass is made to retrace the rollout by reseeding with the rollout's own
    seed and feeding the same inputs. That reproduces the sampling exactly as
    long as nothing else consumed the RNG in between; where it nonetheless
    diverges, the divergence point truncates that rollout's log-prob list
    rather than pairing a new token's gradient with the old token's reward.
    The returned match ratio makes any such drift visible in the training log
    instead of quietly weakening the update.

    Steps inside the sampled context prefix are skipped: the context is
    conditioning, not behaviour under evaluation (Kyutai Sec 3.3).
    """
    group_size = len(rollouts)
    collected: list[list[torch.Tensor]] = [[] for _ in range(group_size)]
    matched = 0
    total = 0
    diverged = [False] * group_size

    _seed_all(seed)
    with lm_gen.streaming(group_size):
        for step, codes in enumerate(user_codes):
            batch = codes.to(device).expand(group_size, -1, -1)
            out = lm_gen.step(batch)
            if out is None:
                continue
            logits = capture.take()
            if step < context_steps:
                continue
            log_probs = torch.log_softmax(logits.float().reshape(group_size, -1), dim=-1)
            ids = out[:, 0, 0].tolist()
            for b in range(group_size):
                if diverged[b]:
                    continue
                index = len(collected[b]) + context_steps
                recorded = rollouts[b]["text_token_ids"]
                if index >= len(recorded):
                    diverged[b] = True
                    continue
                total += 1
                if int(ids[b]) != recorded[index]:
                    diverged[b] = True
                    continue
                matched += 1
                collected[b].append(log_probs[b, recorded[index]])

    if collected and collected[0] and not collected[0][0].requires_grad:
        raise RuntimeError(
            "The re-run produced log probs with no gradient. moshi's LMGen.step "
            "is wrapped in torch.no_grad() in this install, so the GRPO loss "
            "would be a constant. Patch LMGen.step to run under grad for the "
            "update pass (see docs/grpo_judge_lora.md)."
        )
    return collected, (matched / total if total else 0.0)


# -- GRPO loss --------------------------------------------------------

def compute_grpo_loss(
    advantages: list[float],
    old_log_probs: list[list[float]],
    new_log_probs: list[list[torch.Tensor]],
    ref_log_probs: list[list[float]],
    clip_epsilon: float,
    kl_beta: float,
    device: torch.device,
) -> torch.Tensor:
    """Clipped surrogate GRPO loss with KL penalty (Kyutai Eq. 3-4)."""
    losses: list[torch.Tensor] = []

    for i, adv in enumerate(advantages):
        if not old_log_probs[i] or not new_log_probs[i]:
            continue
        min_len = min(len(old_log_probs[i]), len(new_log_probs[i]))
        if min_len == 0:
            continue

        old_lp = torch.tensor(old_log_probs[i][:min_len], device=device)
        new_lp = torch.stack(new_log_probs[i][:min_len]).to(device)

        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate = -torch.min(ratio * adv, clipped * adv).mean()

        kl_loss = torch.zeros((), device=device)
        if ref_log_probs[i]:
            ref_len = min(min_len, len(ref_log_probs[i]))
            ref_lp = torch.tensor(ref_log_probs[i][:ref_len], device=device)
            new_kl = new_lp[:ref_len]
            # k3 estimator: non-negative and lower variance than (new - ref).
            diff = ref_lp - new_kl
            kl = (torch.exp(diff) - diff - 1.0).mean()
            kl_loss = kl_beta * kl

        losses.append(surrogate + kl_loss)

    if not losses:
        return torch.zeros((), device=device, requires_grad=True)
    return torch.stack(losses).mean()


# -- Reward assembly --------------------------------------------------

def build_reward_axes(
    rollouts: list[dict[str, Any]],
    segment: dict[str, Any],
    vad_per_rollout: list[list[tuple[float, float]]],
    judgements: list[dict[str, Any]] | None,
    cfg: GRPOConfig,
) -> tuple[list[list[float]], list[float], list[str]]:
    """Deterministic axis + judge axes, with their weights and names."""
    deterministic = [
        compute_segment_reward(rollout, segment, vad, cfg)
        for rollout, vad in zip(rollouts, vad_per_rollout)
    ]
    axes: list[list[float]] = [deterministic]
    names = [f"det_{segment['axis']}"]
    axis_index = AXES.index(segment["axis"])
    weights = [cfg.reward_weights[axis_index] if axis_index < len(cfg.reward_weights) else 1.0]

    if judgements:
        judge_axes, judge_names = judge_reward_axes(
            judgements, SCORE_KEYS, FLAG_KEYS, cfg.judge_flag_penalty,
        )
        axes.extend(judge_axes)
        names.extend(judge_names)
        for name in judge_names:
            key = name[len("judge_"):]
            weights.append(
                float(cfg.judge_axis_weights.get(key, cfg.judge_weight))
            )
    return axes, weights, names


# -- Main training loop -----------------------------------------------

def train(cfg: GRPOConfig) -> None:
    _seed_all(cfg.seed)
    device = torch.device(cfg.device)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(save_dir / "config.json")

    if not cfg.segment_dir:
        raise ValueError("segment_dir must be set in config")
    seg_datasets = load_segment_datasets(cfg.segment_dir)
    total_segs = sum(len(v) for v in seg_datasets.values())
    for axis, items in seg_datasets.items():
        logger.info("  %s: %d segments", axis, len(items))
    if total_segs == 0:
        raise ValueError(f"No segments found in {cfg.segment_dir}")
    active_axes = [axis for axis in AXES if seg_datasets[axis]]

    logger.info("Loading Moshi to %s ...", device)
    mimi, text_tokenizer, lm_model, checkpoint_info = load_moshi_for_grpo(cfg, device)

    if cfg.use_lora:
        overrides: dict[str, Any] = {}
        if cfg.lora_target_patterns:
            overrides["target_patterns"] = tuple(cfg.lora_target_patterns)
        if cfg.lora_exclude_patterns:
            overrides["exclude_patterns"] = tuple(cfg.lora_exclude_patterns)
        replaced = apply_lora(
            lm_model,
            rank=cfg.lora_rank,
            scaling=cfg.lora_scaling,
            **overrides,
        )
        freeze_base(lm_model)
        logger.info("LoRA: rank=%d scaling=%.2f on %d modules",
                    cfg.lora_rank, cfg.lora_scaling, len(replaced))
        if cfg.lora_resume:
            loaded = load_lora_checkpoint(lm_model, cfg.lora_resume)
            logger.info("Resumed %d adapter tensors from %s", loaded, cfg.lora_resume)
        trainable = lora_parameters(lm_model)
    else:
        for param in lm_model.parameters():
            param.requires_grad = True
        trainable = [p for p in lm_model.parameters() if p.requires_grad]

    n_trainable = sum(p.numel() for p in trainable)
    logger.info("Trainable parameters: %.2fM", n_trainable / 1e6)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )

    capture = TextLogitsCapture(lm_model)
    logger.info("Text logits captured from: %s", capture.module_name)
    lm_gen = _ensure_lm_gen(lm_model, checkpoint_info, cfg)

    judge = judge_from_config(cfg)
    logger.info("Waiting for the judge at %s ...", judge.base_url)
    judge.wait_until_ready(cfg.judge_ready_timeout_sec)
    logger.info("Judge ready: %s", judge.model)

    frame_rate = float(mimi.frame_rate)
    log_path = save_dir / "training_log.jsonl"

    logger.info("***** Starting GRPO training *****")
    logger.info("  Epochs=%d segments/epoch=%d G=%d lr=%s kl_beta=%s",
                cfg.epochs, cfg.segments_per_epoch, cfg.group_size, cfg.lr, cfg.kl_beta)

    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        context_sec = cfg.current_context_sec(epoch)
        logger.info("=== Epoch %d/%d | max_context=%.1fs ===",
                    epoch + 1, cfg.epochs, context_sec)

        epoch_rewards: list[float] = []
        epoch_losses: list[float] = []
        epoch_judge: list[float] = []
        epoch_match: list[float] = []

        rng = random.Random(cfg.seed + epoch)
        epoch_segments: list[tuple[str, dict[str, Any]]] = []
        per_axis = max(1, cfg.segments_per_epoch // len(active_axes))
        for axis in active_axes:
            pool = seg_datasets[axis]
            epoch_segments.extend(
                (axis, s) for s in rng.sample(pool, min(per_axis, len(pool)))
            )
        while len(epoch_segments) < cfg.segments_per_epoch:
            axis = rng.choice(active_axes)
            epoch_segments.append((axis, rng.choice(seg_datasets[axis])))
        rng.shuffle(epoch_segments)

        optimizer.zero_grad(set_to_none=True)

        for s_idx, (axis, segment) in enumerate(epoch_segments):
            logger.info("  Segment %d/%d [%s]: %s", s_idx + 1, len(epoch_segments),
                        axis, Path(segment["wav_path"]).name)

            user_pcm, ctx_offset = load_segment_audio(
                segment, context_sec, cfg.context_prob, rng,
            )
            context_steps = int(ctx_offset * frame_rate)
            rollout_seed = cfg.seed + epoch * 100_003 + s_idx

            user_codes = encode_user_audio(user_pcm, mimi, device)
            if not user_codes:
                logger.warning("    empty audio, skipping")
                continue

            # -- Phase 1: G rollouts, one batched pass --
            lm_model.eval()
            rollouts = generate_group(
                user_codes=user_codes,
                group_size=cfg.group_size,
                lm_gen=lm_gen,
                mimi=mimi,
                text_tokenizer=text_tokenizer,
                capture=capture,
                device=device,
                seed=rollout_seed,
                context_offset_sec=ctx_offset,
            )
            for g, rollout in enumerate(rollouts):
                logger.info("    Rollout %d: %s", g, rollout["transcript"][:50])

            vad_per_rollout = [_vad_on_generated(r["generated_pcm"]) for r in rollouts]

            # -- Phase 2: LLM judge on the Full-Duplex-Bench-JA rubric --
            judgements: list[dict[str, Any]] | None = None
            if cfg.judge_weight > 0 and (
                cfg.judge_all_axes or axis in ("turn_taking", "interruption")
            ):
                rows = [
                    build_packed_row(
                        rollout, segment, vad,
                        row_id=f"e{epoch}_s{s_idx}_g{g}",
                    )
                    for g, (rollout, vad) in enumerate(zip(rollouts, vad_per_rollout))
                ]
                judgements = judge.score_rows(rows)
                overall = [j["scores"]["overall"] for j in judgements]
                epoch_judge.append(sum(overall) / len(overall))

            # -- Phase 3: per-axis decoupled advantages --
            axes, weights, names = build_reward_axes(
                rollouts, segment, vad_per_rollout, judgements, cfg,
            )
            advantages = decoupling_normalize(axes, weights)
            if not any(abs(a) > 1e-9 for a in advantages):
                # Every axis was constant across the group: there is nothing to
                # prefer, and a zero-advantage update is pure wasted compute.
                logger.info("    no reward spread in group, skipping update")
                continue

            det_mean = sum(axes[0]) / len(axes[0])
            logger.info("    R_%s=%.3f%s", axis, det_mean,
                        f" R_judge_overall={epoch_judge[-1]:.2f}" if judgements else "")

            # -- Phase 4: GRPO update on the adapters --
            lm_model.train()
            old_log_probs = [r["text_log_probs"][context_steps:] for r in rollouts]

            # pi_ref is the pre-GRPO policy, i.e. this model with the adapters
            # switched off -- no second copy of Moshi is resident for it.
            with torch.no_grad(), adapters_disabled(lm_model):
                ref_tensors, _ = recompute_log_probs(
                    rollouts, user_codes, lm_gen, capture, device,
                    rollout_seed, context_steps,
                )
            ref_log_probs = [[float(t) for t in row] for row in ref_tensors]

            new_log_probs, match_ratio = recompute_log_probs(
                rollouts, user_codes, lm_gen, capture, device,
                rollout_seed, context_steps,
            )
            epoch_match.append(match_ratio)
            if match_ratio < 0.9:
                logger.warning("    re-run matched only %.1f%% of sampled tokens",
                               match_ratio * 100)

            loss = compute_grpo_loss(
                advantages=advantages,
                old_log_probs=old_log_probs,
                new_log_probs=new_log_probs,
                ref_log_probs=ref_log_probs,
                clip_epsilon=cfg.clip_epsilon,
                kl_beta=cfg.kl_beta,
                device=device,
            )
            (loss / cfg.grad_accum_segments).backward()

            epoch_losses.append(float(loss.item()))
            epoch_rewards.append(det_mean)
            logger.info("    Loss: %.6f", loss.item())

            # The previous version accumulated gradients for the whole run and
            # never called optimizer.step(), so the weights never moved.
            if (s_idx + 1) % cfg.grad_accum_segments == 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if len(epoch_segments) % cfg.grad_accum_segments != 0:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        elapsed = time.time() - epoch_start
        entry = {
            "epoch": epoch + 1,
            "context_sec": context_sec,
            "mean_reward": sum(epoch_rewards) / max(len(epoch_rewards), 1),
            "mean_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
            "mean_judge_overall": (
                sum(epoch_judge) / len(epoch_judge) if epoch_judge else None
            ),
            "mean_token_match": (
                sum(epoch_match) / len(epoch_match) if epoch_match else None
            ),
            "judge_failures": judge.failures,
            "elapsed_sec": round(elapsed, 1),
        }
        logger.info("Epoch %d done in %.1fs | %s", epoch + 1, elapsed,
                    json.dumps({k: v for k, v in entry.items() if k != "epoch"}))
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if (epoch + 1) % cfg.save_every == 0:
            _save(cfg, lm_model, save_dir / f"checkpoint_epoch_{epoch + 1:04d}", epoch + 1)

    _save(cfg, lm_model, save_dir / f"checkpoint_epoch_{cfg.epochs:04d}", cfg.epochs)
    capture.close()
    logger.info("Training complete: %s", save_dir)


def _save(cfg: GRPOConfig, lm_model: nn.Module, out_dir: Path, epoch: int) -> None:
    """Save adapters in the layout scripts/merge_lora.py already consumes."""
    if cfg.use_lora:
        path = save_lora_checkpoint(
            lm_model, out_dir / "consolidated", cfg.lora_rank, cfg.lora_scaling,
            extra={"epoch": epoch, "source": "grpo"},
        )
        logger.info("Saved adapters: %s", path / "lora.safetensors")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(lm_model.state_dict(), out_dir / "model.pt")
        logger.info("Saved checkpoint: %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--segment-dir", type=str, default=None,
                        help="Override segment_dir from config")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Override save_dir from config")
    args = parser.parse_args()
    cfg = GRPOConfig.from_yaml(args.config)

    seg_dir = args.segment_dir or os.environ.get("GRPO_SEGMENT_DIR", "")
    if seg_dir:
        cfg.segment_dir = seg_dir
    save_dir = args.save_dir or os.environ.get("GRPO_SAVE_DIR", "")
    if save_dir:
        cfg.save_dir = save_dir
    # The PBS job passes the merged SFT checkpoint and any adapter to continue
    # from as environment variables, so one config file serves a chain of jobs.
    weight = os.environ.get("MOSHI_WEIGHT", "")
    if weight:
        cfg.moshi_weight = weight
    resume = os.environ.get("LORA_RESUME", "")
    if resume:
        cfg.lora_resume = resume

    train(cfg)


if __name__ == "__main__":
    main()
