# GRPO with LoRA adapters and a local Full-Duplex-Bench-JA judge

GRPO interactivity alignment (Kyutai, arxiv 2606.11167) where the semantic
reward is the same rubric the Azure evaluation uses, served locally by
Qwen3.6-27B on vLLM.

## GPU layout

Two A100 80GB, one PBS job:

| GPU | Contents | Footprint |
| --- | --- | --- |
| 0 | Moshi policy (bf16) + mimi + LoRA adapters + optimizer | well under 40GB |
| 1 | vLLM serving Qwen3.6-27B | ~54GB weights + ~18GB KV cache at `--gpu-memory-utilization 0.90` |

The judge answers over HTTP, so its memory never enters the training process's
allocator. It is resident for the whole job: loading a 27B model takes minutes,
and the previous per-segment load/free could not finish a single epoch.

## Why LoRA

Beyond fitting two GPUs, LoRA is what makes the reference policy free. GRPO's
KL term needs `pi_ref`, and with adapters that is just the same weights with the
adapters switched off (`scripts/grpo/lora.adapters_disabled`). Full fine-tuning
would need a second resident copy of Moshi, and DeepSpeed ZeRO3 -- which
all-gathers every parameter on every forward -- is close to worst-case for
frame-by-frame streaming generation.

Adapters are written as `lora.safetensors` plus a `config.json` carrying
`lora_rank`/`lora_scaling`, i.e. moshi-finetune's layout, so
`scripts/merge_lora.py` and the existing merge -> `response_recorder.py` ->
Full-Duplex-Bench-JA chain consume them unchanged.

Start from the **merged SFT model** (`MOSHI_WEIGHT`), not the base checkpoint.
`pi_ref` is whatever is loaded there, so loading the base model would have the
KL term pulling the policy back toward an untuned listener.

## Reward

`scripts/grpo/judge_prompt.py` re-exports `eval/judge_full_duplex_azure.py`.
The rubric, the flag definitions, the validator and the normalizer are the same
objects the Azure run uses -- editing the rubric there changes the reward here.

Each rollout is reshaped into a Full-Duplex-Bench-JA packed row by
`scripts/grpo/rollout_to_judge_input.py`: segment metadata becomes the user
timeline and the event timeline, the generated text events become timestamped
assistant chunks, and VAD on the generated audio becomes the deterministic
timing metrics. The judge sees no audio, so those metrics are the only thing
telling it when the model actually spoke.

Rewards stay separated per rubric axis (8 scores + one flag-penalty axis, plus
the segment's deterministic axis) and are z-normalized independently within the
group before being weighted and summed. An axis whose group spread is below
`min_std` contributes nothing, which matters because a judge emitting integers
on a 1-5 scale routinely gives all G rollouts the same score.

## Judge separation for reporting

Training against this rubric means Full-Duplex-Bench-JA scores from the same
rubric are no longer an unbiased evaluation of the trained model. Keep the
Azure GPT judge as the held-out judge for anything reported, use Qwen only as
the training reward, and report the correlation between the two.

## If LMGen.step runs under no_grad

The update pass re-runs the group with gradients on and raises if the collected
log probs have no `grad_fn`. Some moshi releases decorate `LMGen.step` with
`@torch.no_grad()`; if that is this install, the GRPO loss would be a constant.
Patch it the way `scripts/patch_kyutai_moshi_finetune.py` patches upstream:
remove the decorator (or add a `grad_enabled` flag) so the update pass can
differentiate through it. Generation still runs inside an explicit
`torch.no_grad()` block, so the patch does not cost rollout memory.

## Known limitation of the update pass

moshi's generator samples internally and offers no token forcing, so the update
pass reproduces the rollout by reseeding with the rollout's seed and replaying
the same inputs. Where a rollout nonetheless diverges, that rollout's log-prob
list is truncated at the divergence point rather than pairing a new token's
gradient with the old token's reward. The per-epoch `mean_token_match` in
`training_log.jsonl` reports how much of the group retraced; a value well below
1.0 means the update is weaker than it looks and the sampling path needs a
proper teacher-forcing hook.

## Submit

One job runs the whole chain: segment extraction (if needed) -> judge server ->
GRPO training -> merge -> Full-Duplex-Bench-JA.

```bash
qsub -V -v BATCH_ID=qwen_dialogues_1000,MOSHI_WEIGHT=$PWD/merged_model/consolidated.safetensors scripts/2026-09-04/grpo_judge_lora.pbs
```

Everything lands under one directory:

```
experiments/grpo/<RUN_NAME>/
  config.yaml                 the exact config this run used
  segments/                   per-axis segment JSONL, if extracted here
  checkpoints/                checkpoint_epoch_XXXX/consolidated/lora.safetensors
  merged/consolidated.safetensors
  eval/                       Full-Duplex-Bench-JA output
  logs/                       train.log, judge_server.log, merge.log, eval.log
  training_log.jsonl          per-epoch reward / loss / judge score
```

`RUN_NAME` (default `grpo_judge_lora`) is a fixed name, not a timestamp. The job
picks up the newest adapter already in `checkpoints/` and continues from it, so
re-submitting the identical command after a 24h walltime kill extends the run
instead of starting a second one beside it. Adapters resume; optimizer state
does not, same as the SFT chain. Pass `RESUME=0` to start clean.

The merge folds the adapter back into `MOSHI_WEIGHT` via the new
`--base-weight` flag on `scripts/merge_lora.py`. This matters: the GRPO adapter
sits on top of the merged SFT model, and merging it into the HF base instead
would discard the fine-tuning underneath it.

Skip stages with `RUN_MERGE=0` or `RUN_EVAL=0`. The evaluation runs last and its
failure is reported but does not fail the job, since the checkpoints and the
merge are already on disk by then.
