"""GRPO hyperparameter configuration.

Reproduces the Kyutai Multi-Faceted Interactivity Alignment paper
(arxiv 2606.11167), adapted for 2x A100 80GB.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class GRPOConfig:
    # --- Model ---
    hf_repo: str = "llm-jp/llm-jp-moshi-v1"
    moshi_weight: str | None = None
    moshi_config: str | None = None
    device: str = "cuda"
    half: bool = False

    # --- LoRA ---
    # GRPO trains adapters only. Beyond fitting two GPUs, this is what makes
    # pi_ref free: the base model with adapters switched off IS the reference
    # policy, so the KL term needs no second resident copy of Moshi.
    use_lora: bool = True
    lora_rank: int = 32                  # SFT uses 128; GRPO only reshapes the
                                         # text-token distribution, and higher
                                         # rank mostly buys instability here.
    lora_scaling: float = 2.0            # alpha = rank * scaling, as in the
                                         # SFT configs under experiments/.
    lora_target_patterns: list[str] = dataclasses.field(default_factory=list)
    # Guards against adapters landing on the depth transformer, whose
    # parameters receive no gradient from a text-token-only GRPO loss.
    lora_exclude_patterns: list[str] = dataclasses.field(default_factory=list)
    lora_resume: str = ""                # adapter to continue from (weights only)

    # --- GRPO core (Kyutai paper Table 1) ---
    # The paper's 2e-7 is a full-fine-tuning learning rate. LoRA updates a
    # low-rank residual initialised at zero and needs roughly two orders of
    # magnitude more, or the adapter never leaves its initialisation.
    lr: float = 1e-5
    epochs: int = 100
    segments_per_epoch: int = 8          # paper=32, reduced for 2xA100
    group_size: int = 8                  # paper G=16, reduced for 2xA100
    kl_beta: float = 0.01
    clip_epsilon: float = 0.2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 2.0
    # Optimizer steps every N segments. Each segment is already a full GRPO
    # group of G rollouts, so 1 is a real step, not a fragment of one.
    grad_accum_segments: int = 1

    # Context length scheduling: 0 → max_context_sec over epochs (Sec 3.3).
    # Context = recording preceding the segment, prepended to input.
    max_context_sec: float = 30.0
    context_prob: float = 0.5            # prob of prepending context

    # --- Segment data ---
    segment_dir: str = ""                # dir with {pause,turn_taking,backchannel,interruption}_segments.jsonl

    # --- Reward axes ---
    pause_threshold_sec: float = 1.0
    backchannel_window_sec: float = 1.0
    backchannel_max_dur_sec: float = 1.0

    # --- LLM judge ---
    # The judge runs behind a resident vLLM OpenAI-compatible server on its own
    # GPU. It scores the Full-Duplex-Bench-JA rubric imported verbatim from
    # eval/judge_full_duplex_azure.py, so the training reward and the reported
    # evaluation cannot drift apart.
    judge_model: str = "Qwen/Qwen3.6-27B"
    judge_base_url: str = "http://127.0.0.1:8000/v1"
    judge_max_tokens: int = 1000         # the rubric reply is a JSON object
                                         # with 8 scores, 4 flags and a reason;
                                         # the old value of 8 truncated it.
    judge_temperature: float = 0.0
    judge_concurrency: int = 8           # one request per rollout in a group
    judge_guided_decoding: bool = True
    judge_ready_timeout_sec: float = 1800.0
    # Applied to every axis, not just turn_taking/interruption: the rubric has
    # axes (backchannel_naturalness, interruption_handling) that speak directly
    # to the other two, and restricting it would leave those axes with only
    # their deterministic timing reward.
    judge_all_axes: bool = True
    judge_flag_penalty: float = 1.0
    # Per-rubric-axis weights, keyed by the SCORE_KEYS names plus "flags".
    # Empty means every axis gets judge_weight.
    judge_axis_weights: dict = dataclasses.field(default_factory=dict)

    # Reward weights per axis [pause, turn, backchannel, interruption].
    # LLM judge weight is separate (added to turn_taking and interruption).
    reward_weights: list[float] = dataclasses.field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0],
    )
    judge_weight: float = 1.0

    # --- Inference ---
    temp: float = 0.7                    # paper: 0.7 for text
    temp_text: float = 0.7
    temp_audio: float = 0.8             # paper: 0.8 for audio
    top_k_audio: int = 250              # paper: 250

    # --- Checkpointing ---
    save_dir: str = "experiments/grpo"
    save_every: int = 5       # save every N epochs
    log_every: int = 1

    # --- Hardware ---
    nproc: int = 2
    seed: int = 42
    disable_triton: bool = False

    # --- Legacy (kept for backward compat, not used in segment-based training) ---
    prompts_file: str = "eval_sets/loneliness_support_prompts.txt"
    eval_cases_file: str = "eval_sets/loneliness_support.jsonl"
    silence_sec: float = 12.0
    max_gen_sec: float = 45.0
    groups_per_epoch: int = 8

    @classmethod
    def from_yaml(cls, path: str | Path) -> GRPOConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return cls()
        fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in fields}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def current_context_sec(self, epoch: int) -> float:
        """Linear context length schedule: 0 → max_context_sec over epochs."""
        if self.epochs <= 1:
            return self.max_context_sec
        frac = min(epoch / (self.epochs - 1), 1.0)
        return frac * self.max_context_sec
