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

    # --- GRPO core (Kyutai paper Table 1) ---
    lr: float = 2e-7
    epochs: int = 100
    groups_per_epoch: int = 8          # paper=32, reduced for 2xA100
    group_size: int = 8                # paper G=16, reduced for 2xA100
    kl_beta: float = 0.01
    clip_epsilon: float = 0.2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 2.0

    # Context length scheduling: linear 0 → max_context_sec over all epochs.
    max_context_sec: float = 30.0

    # --- Reward axes ---
    pause_threshold_sec: float = 1.0
    backchannel_window_sec: float = 1.0
    backchannel_max_dur_sec: float = 1.0

    # LLM judge
    judge_model: str = "Qwen/Qwen3.6-27B"
    judge_max_tokens: int = 256
    judge_temperature: float = 0.0

    # Reward decoupling normalization weights (axis order: pause, turn,
    # backchannel, interruption, judge). Safety weight is folded into judge.
    reward_weights: list[float] = dataclasses.field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0],
    )

    # --- Data ---
    segment_dir: str = ""
    prompts_file: str = "eval_sets/loneliness_support_prompts.txt"
    eval_cases_file: str = "eval_sets/loneliness_support.jsonl"

    # --- Inference ---
    silence_sec: float = 12.0
    max_gen_sec: float = 45.0
    temp: float = 0.8
    temp_text: float = 0.8

    # --- Checkpointing ---
    save_dir: str = "experiments/grpo"
    save_every: int = 5       # save every N epochs
    log_every: int = 1

    # --- Hardware ---
    nproc: int = 2
    seed: int = 0
    disable_triton: bool = False

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
