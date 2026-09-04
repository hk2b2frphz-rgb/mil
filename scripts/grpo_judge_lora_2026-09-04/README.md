# grpo_judge_lora (2026-09-04)

GRPO interactivity alignment on LoRA adapters, with a local Qwen3.6-27B judge
scoring the Full-Duplex-Bench-JA rubric. See `docs/grpo_judge_lora.md`.

## Run it

One job. Nothing has to run first.

```bash
qsub -V scripts/grpo_judge_lora_2026-09-04/30_grpo_train.pbs
```

`30_grpo_train.pbs` extracts GRPO segments from an already rendered corpus, then
runs judge -> GRPO -> merge -> Full-Duplex-Bench-JA.

| File | Queue | Role |
| --- | --- | --- |
| `30_grpo_train.pbs` | `xan_s` / `res=middle` | **entry point**: segments -> GRPO -> merge -> eval |
| `10_dialogue.pbs` | `xan_s` / `res=small` | optional: 10,000 multi-agent dialogues |
| `20_tts.pbs` | `xvn_s` / `res=middle2` | optional: Qwen3-TTS render to stereo WAV |

Every argument is hardcoded. These files are the experiment record, so there is
nothing to reconstruct from a submit shell's environment later. To run a
different corpus or version, copy the folder and edit the constants.

## Why stages 1 and 2 are optional

GRPO touches `segments_per_epoch * epochs = 8 * 100 = 800` segments over the
whole run, round-robined across four axes — roughly **200 per axis**.
`segment_extractor.py` caps extraction at **2000 per axis**, and even a
2,000-dialogue corpus saturates that cap. Generating 10,000 fresh dialogues and
rendering them would spend days of TTS on an order of magnitude more data than
the run can consume.

So `30_grpo_train.pbs` points `CORPUS_ROOT` at an existing corpus
(`data/runs/response_2000_v1` by default). Run stages 1 and 2 only if no
rendered response corpus exists, then edit `CORPUS_ROOT` to point at the output.
Stage 2 deliberately does not chain into training: which corpus gets trained on
is an edit, not something a chained job should decide.

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
data/runs/response_2000_v1/
  tts/merged/training_set/synthetic_moshi_train.jsonl   input corpus
  grpo_segments/{pause,turn_taking,backchannel,interruption}_segments.jsonl

experiments/grpo/grpo_response_2000_v1/
  config.yaml  checkpoints/  merged/  eval/  logs/  training_log.jsonl
```

## Prerequisite

`merged_model/consolidated.safetensors` — the merged SFT model. GRPO's reference
policy is that model with the adapters switched off, so stage 3 stops with an
explicit error rather than aligning the untuned base by accident.

```bash
qsub -v LORA_CKPT=<sft>/consolidated/lora.safetensors,OUT_WEIGHT=$PWD/merged_model/consolidated.safetensors scripts/merge_lora.pbs
```

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
