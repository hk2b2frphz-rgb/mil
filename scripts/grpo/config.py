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
    segments_per_epoch: int = 8          # paper=32, reduced for 2xA100
    group_size: int = 8                  # paper G=16, reduced for 2xA100
    kl_beta: float = 0.01
    clip_epsilon: float = 0.2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 2.0

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

    # LLM judge (applied to turn_taking and interruption only, per paper)
    judge_model: str = "Qwen/Qwen3.6-27B"
    judge_max_tokens: int = 8
    judge_temperature: float = 0.0

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
