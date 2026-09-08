# grpo_judge_lora (2026-09-04)

GRPO interactivity alignment on LoRA adapters, with a local Qwen3.6-27B judge
scoring the Full-Duplex-Bench-JA rubric. See `docs/grpo_judge_lora.md`.

## Run it

Three stages, chained. Submit stage 1 and the rest follow.

```bash
qsub -V scripts/grpo_judge_lora_2026-09-04/10_dialogue.pbs
```

| Stage | File | Queue | What it does |
| --- | --- | --- | --- |
| 1 | `10_dialogue.pbs` | `xan_s` / `res=small` | 2,000 multi-agent dialogues |
| 2 | `20_tts.pbs` | `xvn_s` / `res=middle2` | Qwen3-TTS render to stereo WAV |
| 3 | `30_grpo_train.pbs` | `xan_s` / `res=middle` | segments -> GRPO -> merge -> eval |

If a rendered multi-agent corpus already exists, skip stages 1 and 2: point
`CORPUS_ROOT` in `30_grpo_train.pbs` at it and submit that file alone.

Experiment settings are hardcoded. To run a different corpus or version, copy
the folder and edit the constants. Stage 3 accepts SFT input/output settings
through the environment as described below; selected paths are printed in the job log.

## Why 2,000 dialogues and not 10,000

GRPO touches `segments_per_epoch * epochs = 8 * 100 = 800` segments over the
whole run, round-robined across four axes — roughly **200 per axis**.
`segment_extractor.py` caps extraction at **2000 per axis**, and 2,000 dialogues
at 5-8 pairs already saturates that cap. Rendering five times more would spend
days of TTS on data the run cannot reach.

If the run ever needs materially more exposure, raise `epochs` first and read
the per-axis segment counts stage 3 prints — generating more data is the last
resort, not the first.

## The corpus must be multi-agent

`DIALOGUE_GENERATION_MODE=multi-agent`, the response-style generation used
before the aizuchi model work, where moshi actually answers.

Not the aizuchi corpus: that runs `DIALOGUE_GENERATION_MODE=aizuchi-only`, which
pushes every dialogue through `sanitize_aizuchi_only_turns` and mechanically
drops any moshi utterance outside the backchannel vocabulary. GRPO extracts four
segment types (pause, turn-taking, backchannel, interruption); on an
aizuchi-only corpus three of those four are empty.

Backchannels still have to be present, though: `gt_backchannel_times` for the
backchannel axis is read off the moshi channel of these recordings.

Stage 3 prints the per-axis segment count before training starts. If any axis is
at or near zero, the corpus is wrong and training would silently optimise three
axes instead of four.

## Paths

```
data/runs/grpo_response_2000_v1/
  dialogue/llm_dialogues/dialogues.jsonl
  tts/merged/training_set/synthetic_moshi_train.jsonl
  grpo_segments/{pause,turn_taking,backchannel,interruption}_segments.jsonl

experiments/grpo/grpo_response_2000_v1/
  config.yaml  checkpoints/  merged/  eval/  logs/  training_log.jsonl
```

## SFT merge inside the training job

Stage 3 reuses `merged_model/consolidated.safetensors` if it is a nonempty file.
Otherwise it merges `SFT_LORA_CKPT` before segment extraction and GRPO, inside
the same PBS job. No separate merge submission is needed. GRPO's reference
policy is this merged SFT model with the GRPO adapters switched off.

```bash
qsub -v SFT_LORA_CKPT=/path/to/sft/consolidated/lora.safetensors scripts/grpo_judge_lora_2026-09-04/30_grpo_train.pbs
```

Replace the adapter path with the actual SFT checkpoint. Optional environment
settings: `MOSHI_WEIGHT` (merged output path), `MOSHI_FT_REPO` (defaults to
`../moshi-finetune`), `SFT_HF_REPO` (defaults to `llm-jp/llm-jp-moshi-v1`), and
`SFT_LORA_SCALING` (otherwise read by the merge script from the adapter config).
Use the same base repository and scaling as SFT. An existing merged model takes
precedence over the adapter setting; select a new `MOSHI_WEIGHT` and a new run
when changing the SFT model. A failed merge stops the job, and output is published
only after the merge succeeds so an interrupted write is not reused on retry.

## Resume

Re-submit the same file. Stage 3 picks up the newest adapter under
`experiments/grpo/grpo_response_2000_v1/checkpoints/`. Adapters resume,
optimizer state does not — same as the SFT chain.

## Two GPUs

Stage 3 needs both: GPU 0 trains the policy, GPU 1 holds the 27B judge behind a
vLLM server. The judge is stopped as soon as training ends so the merge and the
evaluation get the whole node.

The automatic evaluation at the end is judged by the **same rubric GRPO trained
against**, so treat it as a training-progress signal. Anything reported should be
judged by Azure GPT (`eval/judge_full_duplex_azure.py`) as a held-out judge.
