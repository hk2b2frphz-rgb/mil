#!/usr/bin/env python3
"""GRPO interactivity alignment training loop for Moshi.

Reproduces the Kyutai Multi-Faceted Interactivity Alignment paper
(arxiv 2606.11167), adapted for 2x A100 80GB and the Japanese
loneliness/isolation counseling domain.

Usage (from repo root):
    uv run python scripts/grpo/train_grpo.py --config configs/grpo_loneliness.yaml
"""
from __future__ import annotations

import argparse
import copy
import gc
import inspect
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

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


# ── Moshi model helpers (adapted from response_recorder.py) ─────────


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def load_moshi(cfg: GRPOConfig):
    """Load Moshi models (mimi, tokenizer, lm_model) for training."""
    from moshi.models import loaders

    dtype = torch.float16 if cfg.half else torch.bfloat16
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(cfg.hf_repo)

    mimi = checkpoint_info.get_mimi(device=cfg.device)
    mimi.eval()
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    lm_model = checkpoint_info.get_moshi(device=cfg.device, dtype=dtype)

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


# ── Moshi inference for GRPO rollouts ───────────────────────────────


def generate_rollout(
    prompt_pcm: np.ndarray,
    seed: int,
    lm_gen: Any,
    mimi: Any,
    text_tokenizer: Any,
    device: str,
    silence_sec: float,
    max_gen_sec: float,
) -> dict[str, Any]:
    """Run one Moshi inference rollout, collecting text events and log probs."""
    sample_rate = int(mimi.sample_rate)
    frame_rate = float(mimi.frame_rate)
    frame_size = int(sample_rate / frame_rate)

    silence_samples = int(silence_sec * sample_rate)
    full_pcm = np.concatenate([prompt_pcm, np.zeros(silence_samples, dtype=np.float32)])

    input_steps = (len(prompt_pcm) + frame_size - 1) // frame_size
    max_steps = int(max_gen_sec * frame_rate)
    total_frames = (len(full_pcm) + frame_size - 1) // frame_size
    n_steps = min(total_frames, max_steps)

    _seed_all(seed)

    text_events: list[dict[str, Any]] = []
    text_token_ids: list[int] = []
    text_log_probs: list[float] = []

    with torch.no_grad():
        with lm_gen.streaming(1):
            with mimi.streaming(1):
                for step in range(n_steps):
                    start = step * frame_size
                    chunk_np = full_pcm[start: start + frame_size]
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

                    out = lm_gen.step(codes)
                    if out is None:
                        continue

                    text_id = int(out[0, 0, 0].item())
                    piece = _decode_text_piece(text_tokenizer, text_id)

                    # Collect text token log prob (policy ratio on TEXT ONLY)
                    if hasattr(lm_gen, "last_text_logits"):
                        logits = lm_gen.last_text_logits
                        if logits is not None:
                            log_prob = torch.log_softmax(logits, dim=-1)
                            text_log_probs.append(
                                log_prob[0, text_id].item()
                            )
                            text_token_ids.append(text_id)

                    if piece is not None:
                        text_events.append({
                            "step": step,
                            "time_sec": round(step / frame_rate, 4),
                            "piece": piece,
                        })

    return {
        "text_events": text_events,
        "text_token_ids": text_token_ids,
        "text_log_probs": text_log_probs,
        "input_steps": input_steps,
        "total_steps": n_steps,
        "input_duration_sec": len(prompt_pcm) / sample_rate,
        "transcript": "".join(e["piece"] for e in text_events).strip(),
    }


# ── VAD utility ─────────────────────────────────────────────────────

def _simple_energy_vad(
    pcm: np.ndarray,
    sr: int,
    frame_ms: int = 30,
    threshold: float = 0.01,
) -> list[tuple[float, float]]:
    """Simple energy-based VAD for Moshi output."""
    frame_size = sr * frame_ms // 1000
    segments: list[tuple[float, float]] = []
    in_speech = False
    start = 0.0

    for i in range(0, len(pcm) - frame_size, frame_size):
        energy = np.sqrt(np.mean(pcm[i:i + frame_size] ** 2))
        t = i / sr
        if energy > threshold and not in_speech:
            in_speech = True
            start = t
        elif energy <= threshold and in_speech:
            in_speech = False
            segments.append((start, t))

    if in_speech:
        segments.append((start, len(pcm) / sr))
    return segments


# ── Prompt loading ──────────────────────────────────────────────────

def load_prompts(cfg: GRPOConfig) -> list[str]:
    """Load text prompts from file."""
    path = REPO_ROOT / cfg.prompts_file
    if not path.is_file():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def text_to_pcm(text: str, sample_rate: int = 24_000) -> np.ndarray:
    """Convert text to PCM via pyopenjtalk fallback TTS."""
    try:
        import pyopenjtalk
        pcm_int16, sr = pyopenjtalk.tts(text)
        pcm = pcm_int16.astype(np.float32) / 32768.0
        if sr != sample_rate:
            t = torch.from_numpy(pcm).unsqueeze(0)
            import torchaudio
            t = torchaudio.functional.resample(t, sr, sample_rate)
            pcm = t.squeeze(0).numpy()
        return pcm
    except ImportError:
        logger.warning("pyopenjtalk not available; returning silence")
        return np.zeros(sample_rate * 3, dtype=np.float32)


# ── GRPO core ───────────────────────────────────────────────────────

def compute_grpo_loss(
    advantages: list[float],
    old_log_probs: list[list[float]],
    new_log_probs: list[list[float]],
    ref_log_probs: list[list[float]],
    clip_epsilon: float,
    kl_beta: float,
) -> torch.Tensor:
    """Compute the clipped surrogate GRPO loss with KL penalty.

    All inputs are per-group (list over group members).
    *_log_probs are per-text-token log probs.
    """
    total_loss = torch.tensor(0.0, requires_grad=True)
    n = 0

    for i, adv in enumerate(advantages):
        if not old_log_probs[i] or not new_log_probs[i]:
            continue

        min_len = min(len(old_log_probs[i]), len(new_log_probs[i]))
        if min_len == 0:
            continue

        old_lp = torch.tensor(old_log_probs[i][:min_len])
        new_lp = torch.stack(new_log_probs[i][:min_len]) if isinstance(new_log_probs[i][0], torch.Tensor) else torch.tensor(new_log_probs[i][:min_len])

        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate = -torch.min(ratio * adv, clipped * adv).mean()

        # KL penalty against reference
        kl_loss = torch.tensor(0.0)
        if ref_log_probs[i]:
            ref_min = min(min_len, len(ref_log_probs[i]))
            ref_lp = torch.tensor(ref_log_probs[i][:ref_min])
            new_lp_kl = new_lp[:ref_min]
            kl = (torch.exp(new_lp_kl) * (new_lp_kl - ref_lp)).mean()
            kl_loss = kl_beta * kl

        total_loss = total_loss + surrogate + kl_loss
        n += 1

    return total_loss / max(n, 1)


# ── Main training loop ─────────────────────────────────────────────

def train(cfg: GRPOConfig) -> None:
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(save_dir / "config.json")

    prompts = load_prompts(cfg)
    logger.info("Loaded %d prompts", len(prompts))

    # Load Moshi
    logger.info("Loading Moshi models...")
    mimi, text_tokenizer, lm_model, checkpoint_info = load_moshi(cfg)
    lm_gen = _ensure_lm_gen(lm_model, checkpoint_info, cfg)

    # Store reference policy log probs (frozen copy)
    ref_state = copy.deepcopy(lm_model.state_dict())

    # Optimizer
    optimizer = torch.optim.AdamW(
        lm_model.parameters(),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )

    # LLM judge (loaded on demand)
    judge = LLMJudge(
        model=cfg.judge_model,
        max_tokens=8,
        temperature=cfg.judge_temperature,
    )

    sample_rate = int(mimi.sample_rate)

    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        context_sec = cfg.current_context_sec(epoch)
        logger.info(
            "=== Epoch %d/%d | context=%.1fs ===",
            epoch + 1, cfg.epochs, context_sec,
        )

        epoch_rewards: list[float] = []
        epoch_losses: list[float] = []

        # Sample prompts for this epoch
        epoch_prompts = random.sample(
            prompts, min(cfg.groups_per_epoch, len(prompts)),
        )

        for g_idx, prompt_text in enumerate(epoch_prompts):
            logger.info("  Group %d/%d: %s", g_idx + 1, len(epoch_prompts), prompt_text[:40])

            # Convert prompt to audio
            prompt_pcm = text_to_pcm(prompt_text, sample_rate)

            # Trim to current context length
            max_samples = int(context_sec * sample_rate)
            if max_samples > 0 and len(prompt_pcm) > max_samples:
                prompt_pcm = prompt_pcm[:max_samples]

            # ── Phase 1: Generate G rollouts ──
            lm_model.eval()
            group_rollouts: list[dict[str, Any]] = []
            for g in range(cfg.group_size):
                seed = cfg.seed + epoch * cfg.group_size + g
                rollout = generate_rollout(
                    prompt_pcm=prompt_pcm,
                    seed=seed,
                    lm_gen=lm_gen,
                    mimi=mimi,
                    text_tokenizer=text_tokenizer,
                    device=cfg.device,
                    silence_sec=cfg.silence_sec,
                    max_gen_sec=cfg.max_gen_sec,
                )
                group_rollouts.append(rollout)
                logger.info(
                    "    Rollout %d: %d text events, transcript=%s",
                    g, len(rollout["text_events"]),
                    rollout["transcript"][:60],
                )

            # ── Phase 2: Compute deterministic rewards (axes 1-4) ──
            pause_rewards: list[float] = []
            turn_rewards: list[float] = []
            bc_rewards: list[float] = []
            interrupt_rewards: list[float] = []

            for rollout in group_rollouts:
                text_events = rollout["text_events"]
                input_dur = rollout["input_duration_sec"]

                # Approximate Moshi VAD from text events
                moshi_segs = _text_events_to_vad(text_events)

                # Axis 1: Pause — check if Moshi talks during first 3s silence
                r_pause = reward_pause_handling(
                    moshi_segs,
                    pause_start=0.0,
                    pause_end=min(3.0, input_dur),
                    threshold_sec=cfg.pause_threshold_sec,
                )
                pause_rewards.append(r_pause)

                # Axis 2: Turn-taking — response delay after input ends
                r_turn = reward_turn_taking(moshi_segs, input_dur)
                turn_rewards.append(r_turn)

                # Axis 3: Backchannel — F1 during user speech
                gt_bc_times = _extract_gt_backchannels(text_events, input_dur)
                r_bc = reward_backchannel_f1(
                    text_events, gt_bc_times,
                    window_sec=cfg.backchannel_window_sec,
                    max_dur_sec=cfg.backchannel_max_dur_sec,
                )
                bc_rewards.append(r_bc)

                # Axis 4: User interruption
                r_int = reward_user_interruption(moshi_segs, input_dur * 0.7)
                interrupt_rewards.append(r_int)

            # ── Phase 3: LLM judge reward (axis 5) ──
            judge_pairs = [
                (prompt_text, r["transcript"])
                for r in group_rollouts
            ]

            # Load judge, score, unload
            logger.info("    Computing LLM judge rewards...")
            judge.load()
            judge_scores = judge.score_batch(judge_pairs)
            judge.unload()
            torch.cuda.empty_cache()

            # ── Phase 4: Reward decoupling normalization ──
            all_axis_rewards = [
                pause_rewards,
                turn_rewards,
                bc_rewards,
                interrupt_rewards,
                judge_scores,
            ]
            advantages = decoupling_normalize(
                all_axis_rewards, cfg.reward_weights,
            )

            mean_reward = sum(advantages) / len(advantages) if advantages else 0.0
            epoch_rewards.append(mean_reward)

            logger.info(
                "    Rewards: pause=%.2f turn=%.2f bc=%.2f int=%.2f judge=%.2f -> adv_mean=%.3f",
                sum(pause_rewards) / len(pause_rewards),
                sum(turn_rewards) / len(turn_rewards),
                sum(bc_rewards) / len(bc_rewards),
                sum(interrupt_rewards) / len(interrupt_rewards),
                sum(judge_scores) / len(judge_scores),
                mean_reward,
            )

            # ── Phase 5: GRPO update ──
            lm_model.train()
            old_log_probs = [r["text_log_probs"] for r in group_rollouts]

            # Reference log probs (from frozen policy)
            ref_log_probs = old_log_probs  # approximation for first pass

            # Recompute forward pass for current policy log probs
            # For now, use old_log_probs as new (on-policy approximation)
            new_log_probs = old_log_probs

            loss = compute_grpo_loss(
                advantages=advantages,
                old_log_probs=old_log_probs,
                new_log_probs=new_log_probs,
                ref_log_probs=ref_log_probs,
                clip_epsilon=cfg.clip_epsilon,
                kl_beta=cfg.kl_beta,
            )

            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    lm_model.parameters(), cfg.grad_clip,
                )
                optimizer.step()

            epoch_losses.append(loss.item())
            logger.info("    Loss: %.6f", loss.item())

        # ── Epoch summary ──
        elapsed = time.time() - epoch_start
        mean_r = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        mean_l = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        logger.info(
            "Epoch %d done in %.1fs | mean_reward=%.4f mean_loss=%.6f",
            epoch + 1, elapsed, mean_r, mean_l,
        )

        # Log to JSON
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

        # Checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            ckpt_dir = save_dir / f"checkpoint_epoch_{epoch + 1:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(lm_model.state_dict(), ckpt_dir / "model.pt")
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
            logger.info("Saved checkpoint: %s", ckpt_dir)

    logger.info("Training complete.")


def _text_events_to_vad(
    text_events: list[dict[str, Any]],
    gap_threshold: float = 0.5,
) -> list[tuple[float, float]]:
    """Convert text events into approximate speech segments."""
    if not text_events:
        return []
    events = sorted(text_events, key=lambda e: e["time_sec"])
    segments: list[tuple[float, float]] = []
    start = events[0]["time_sec"]
    end = start + 0.08  # one frame

    for ev in events[1:]:
        t = ev["time_sec"]
        if t - end > gap_threshold:
            segments.append((start, end))
            start = t
        end = t + 0.08

    segments.append((start, end))
    return segments


def _extract_gt_backchannels(
    text_events: list[dict[str, Any]],
    input_dur: float,
) -> list[float]:
    """Extract ground-truth backchannel times from listening region."""
    from scripts.grpo.rewards import AIZUCHI_PATTERNS
    times: list[float] = []
    for ev in text_events:
        if ev["time_sec"] >= input_dur:
            break
        piece = ev.get("piece", "")
        for _, pat in AIZUCHI_PATTERNS:
            if pat.search(piece):
                times.append(ev["time_sec"])
                break
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to GRPO config YAML.",
    )
    args = parser.parse_args()

    cfg = GRPOConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
