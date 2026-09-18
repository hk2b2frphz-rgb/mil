"""LoRA adapters for GRPO, checkpoint-compatible with scripts/merge_lora.py.

Why not import kyutai moshi-finetune's LoRALinear directly: GRPO needs the
reference policy pi_ref for its KL term, and with LoRA that is simply the model
with the adapters switched off. moshi-finetune's layer has no off switch (SFT
never needs one), and monkey-patching its forward is a fragile way to get one.
So the layer is reimplemented here with an ``enabled`` flag -- and with exactly
the same submodule names, ``lora_A`` and ``lora_B``, so the checkpoints this
writes are consumed unchanged by scripts/merge_lora.py and therefore by the
whole existing merge -> response_recorder -> Full-Duplex-Bench-JA eval chain.

Switching adapters off costs one boolean per forward and no extra memory, which
is the reason LoRA fits this job on two GPUs at all: full fine-tuning would need
a second frozen copy of Moshi resident alongside the policy.
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

# Attention and MLP projections of the temporal transformer, and nothing else.
#
# This is not a memory-saving choice, it is what the loss allows. GRPO here
# differentiates the log-probability of the sampled TEXT token (Kyutai Sec 3.4);
# the depth transformer and the audio heads sit downstream of that token and
# contribute exactly zero gradient to it. Adapters placed there would be
# optimizer state that never receives a gradient -- dead parameters that make a
# run look like it is training more than it is.
DEFAULT_TARGET_PATTERNS = (
    r"(^|\.)transformer\.layers\.\d+\.self_attn\.(in_proj|out_proj|q_proj|k_proj|v_proj)$",
    r"(^|\.)transformer\.layers\.\d+\.gating\.(linear_in|linear_out)$",
)

# The depth transformer is a StreamingTransformer too, so depending on the moshi
# release its submodules are named either "depformer.layers.N..." or
# "depformer.transformer.layers.N...". The second form would be caught by the
# patterns above -- exactly the dead-parameter case they exist to avoid -- so it
# is excluded by name rather than left to the pattern anchors.
DEFAULT_EXCLUDE_PATTERNS = (r"depformer", r"depth_transformer")


class LoRALinear(nn.Module):
    """``y = W x + scaling * B(A(x))``, with W frozen and the adapter toggleable."""

    def __init__(self, base: nn.Linear, rank: int, scaling: float) -> None:
        super().__init__()
        self.rank = rank
        self.scaling = scaling
        self.enabled = True

        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        device = base.weight.device
        # The adapter is kept in fp32 while the frozen base may be bf16: the
        # GRPO update is a handful of tiny steps at a small learning rate, and
        # bf16's ~3 decimal digits silently round most of them to zero.
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, device=device,
                                dtype=torch.float32)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, device=device,
                                dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        # B starts at zero, so the adapted model begins identical to the base.
        # For GRPO that is not cosmetic: at step 0 the policy must equal pi_ref
        # or the KL term starts out non-zero against a policy that never
        # produced the rollouts.
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.linear(x, self.weight, self.bias)
        if not self.enabled:
            return out
        delta = self.lora_B(self.lora_A(x.float())) * self.scaling
        return out + delta.to(out.dtype)


def _iter_linear(model: nn.Module) -> Iterator[tuple[str, nn.Module, str, nn.Linear]]:
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                full = f"{name}.{child_name}" if name else child_name
                yield full, module, child_name, child


def apply_lora(
    model: nn.Module,
    rank: int = 32,
    scaling: float = 2.0,
    target_patterns: tuple[str, ...] = DEFAULT_TARGET_PATTERNS,
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS,
) -> list[str]:
    """Replace matching nn.Linear modules with LoRALinear. Returns their names.

    Raises if nothing matched. A silent zero-adapter run would train no
    parameters at all while reporting a perfectly normal-looking loss, so the
    error lists the Linear modules that do exist -- that list is what the
    ``lora_target_patterns`` config entry should be set from.
    """
    compiled = [re.compile(p) for p in target_patterns]
    excluded = [re.compile(p) for p in exclude_patterns]
    replaced: list[str] = []
    skipped: list[str] = []
    candidates: list[str] = []

    for full, parent, child_name, child in list(_iter_linear(model)):
        candidates.append(full)
        if not any(p.search(full) for p in compiled):
            continue
        if any(p.search(full) for p in excluded):
            skipped.append(full)
            continue
        setattr(parent, child_name, LoRALinear(child, rank, scaling))
        replaced.append(full)

    if skipped:
        print(
            f"[lora] excluded {len(skipped)} matched modules "
            f"(e.g. {skipped[0]}): downstream of the text token, no gradient."
        )
    if not replaced:
        sample = "\n  ".join(sorted(set(candidates))[:60])
        raise RuntimeError(
            "LoRA matched no module. Patterns tried:\n  "
            + "\n  ".join(target_patterns)
            + "\nLinear modules present (first 60):\n  "
            + sample
        )
    return replaced


def freeze_base(model: nn.Module) -> None:
    """Freeze everything except the adapters."""
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A.weight, module.lora_B.weight])
    return params


@contextmanager
def adapters_disabled(model: nn.Module) -> Iterator[None]:
    """Run the base (pre-GRPO) policy -- this is pi_ref for the KL term."""
    layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    previous = [m.enabled for m in layers]
    for layer in layers:
        layer.enabled = False
    try:
        yield
    finally:
        for layer, state in zip(layers, previous):
            layer.enabled = state


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Adapter tensors under the key names scripts/merge_lora.py expects."""
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return state


def save_lora_checkpoint(
    model: nn.Module,
    out_dir: str | Path,
    rank: int,
    scaling: float,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``lora.safetensors`` + ``config.json`` in moshi-finetune's layout.

    scripts/merge_lora.py reads ``lora_rank``/``lora_scaling`` as top-level keys
    of config.json, so they are written that way and the checkpoint drops
    straight into the existing merge and evaluation chain.
    """
    import safetensors.torch

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(lora_state_dict(model), str(path / "lora.safetensors"))
    config = {"lora": True, "lora_rank": rank, "lora_scaling": scaling}
    if extra:
        config.update(extra)
    (path / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return path


def load_lora_checkpoint(model: nn.Module, lora_path: str | Path) -> int:
    """Load adapter weights saved by save_lora_checkpoint. Returns tensors loaded."""
    import safetensors.torch

    state = safetensors.torch.load_file(str(lora_path))
    own = dict(model.named_modules())
    loaded = 0
    for key, tensor in state.items():
        if not key.endswith(".weight"):
            continue
        # "....q_proj.lora_A.weight" -> owner "....q_proj", attr "lora_A"
        adapter_path = key[: -len(".weight")]
        owner_name, _, attr = adapter_path.rpartition(".")
        owner = own.get(owner_name)
        if not isinstance(owner, LoRALinear) or attr not in ("lora_A", "lora_B"):
            continue
        target = getattr(owner, attr)
        target.weight.data.copy_(tensor.to(target.weight.dtype))
        loaded += 1
    if loaded == 0:
        raise RuntimeError(f"No adapter tensors from {lora_path} matched this model.")
    return loaded
