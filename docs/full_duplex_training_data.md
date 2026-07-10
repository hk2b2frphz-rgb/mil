# Japanese full-duplex training data

The 350 fixed cases in `eval_sets/full_duplex_ja/scenarios_expanded.jsonl` are evaluation
inputs. They are not used as training text.

Full-duplex training data is generated independently through the repository's
existing pipeline:

1. `build_full_duplex_training_use_cases.py` creates varied support-window use
   cases. By default 70% are free-form chat/listening cards (`duplex_task` is
   null) and 30% are spread over the seven labeled tasks. Diversity comes from
   combining situation, topic, conversation type, persona occupation, time of
   day, personality (12 temperaments), and emotional state axes
   (`build_use_cases.py`). The emotional state biases per-turn emotion labels,
   which map to Qwen3-TTS delivery instructions (e.g. tearful/sobbing,
   high-tension, withdrawn) via `EMOTION_PRESETS`.
2. `build_content_seeds.py` (optional, run once) uses Gemma to brainstorm a
   bank of concrete, mundane talking points (e.g. "the bakery is closing this
   month"). 1-2 are sampled into each card so the actual spoken content varies
   instead of collapsing onto a few stock phrases.
3. `generate_synthetic_moshi_training_data.py` asks Gemma to write Japanese
   counselor dialogues and timing events. The prompt now carries the persona's
   personality, today's emotional state, and the sampled talking points, tells
   Gemma to vary the opening/phrasing, and to keep emotion gradual rather than
   swinging every turn (no forced positive resolution). Sampling uses
   `top_p=0.95`.
4. `enrich_dialogue_timing.py` post-processes dialogues in two passes:
   - Emotion smoothing (all dialogues, labels only): user emotions are grouped
     (distress / anxious / positive / neutral) and constrained to adjacent
     transitions with inertia, so a sobbing turn cannot flip to high-tension the
     next turn. This only edits emotion labels, so task validators are safe.
   - Backchannel injection (free-form dialogues only): long user turns are split
     at clause boundaries and short aizuchi are layered in during the user's
     speech, teaching frequent, fast backchannels instead of one slow one per
     turn. Labeled-task structure is left untouched.
5. `generate_qwen3_tts_data.py` renders stereo audio with Moshi on the left and
   the user/environment on the right. A short transition gap (`--gap-sec 0.2`)
   reflects natural Japanese turn-taking. Unless `--no-opening-greeting` is
   passed, every dialogue's turns are prepended with a fixed Moshi turn
   (`OPENING_GREETING_TEXT`, default
   `"もしもし、こちら孤独孤立相談窓口になります。"`, override with
   `--opening-greeting TEXT`). Moshi generating this line itself every time --
   rather than some external layer playing a canned clip -- is the point:
   emitting it seeds "loneliness/isolation counseling window" as grounding
   context in the model's own generation history before anything else in the
   session happens. The greeting *audio* is synthesized exactly once per run
   with a fixed voice and fixed style instruct
   (`--speaker-moshi` + `--opening-greeting-instruct`), disk-cached under
   `data/.cache/opening_greeting/` (`--opening-greeting-cache-dir`), and the
   identical waveform is reused in every sample -- the per-dialogue style
   preset never touches it, so the model memorizes one consistent way of
   saying its opening line. The insertion happens after
   `validate_duplex_dialogue()` already ran on the un-prefixed turns, so
   per-task timing validators (e.g. "`model_backchannel` must overlap a
   preceding user turn") are unaffected.
   See [`full_duplex_evaluation.md`](full_duplex_evaluation.md) for how the
   eval side reserves matching lead-in time and checks the greeting was
   actually produced.
6. The usual Moshi fine-tuning launchers consume
   `training_set/synthetic_moshi_train.jsonl`.

## Large-scale (~100h) generation

`run_full_duplex_training_data_100h.pbs` is a PBS job array. Each sub-job is a
self-contained shard (use_cases -> dialogues -> enrich -> audio) on one V100,
writing to its own directory. With eight V100s the scheduler runs eight shards
at a time and queues the rest, so no special throttling is needed. Self-
contained shards are resumable: just re-submit failed indices.

Minimal config to reach ~100h on 8x V100:

- 10 shards x 250 dialogues = 2,500 dialogues (~100h at ~150s average).
- If real RTF is slower, extend `walltime` or lower `DIALOGUES_PER_SHARD`.
- Want more? Scale the array up (e.g. `-J 0-95` for ~1000h).

```bash
# Always run one pilot shard first to measure real throughput and the actual
# average dialogue length before committing all shards.
qsub -J 0-0 scripts/run_full_duplex_training_data_100h.pbs

# Full run.
qsub scripts/run_full_duplex_training_data_100h.pbs

# After all shards finish, merge into one manifest (CPU only, login node):
BATCH_ID=<printed_BATCH_ID>
uv run python scripts/merge_training_shards.py \
  --batch-dir data/runs/$BATCH_ID \
  --out-dir   data/runs/$BATCH_ID/merged

# Fine-tune with the merged dataset:
SRC_RUN_DIR=data/runs/$BATCH_ID/merged qsub scripts/fullft_sweep.pbs
```

The merge writes absolute wav paths, so no audio is copied. Useful overrides:
`BATCH_ID`, `RUN_STAMP`, `DIALOGUES_PER_SHARD`, `BASE_SEED`,
`LISTENING_RATIO`, `GAP_SEC`. Default output names avoid clobbering existing
runs: non-array jobs get a timestamp suffix, while array jobs use the PBS array
job id so all shards share one batch directory. Set `RUN_STAMP=YYYYMMDD_HHMMSS`
or `BATCH_ID=...` before submission if you want a human-chosen batch name.

## gpt-oss-120b dialogue generation on 2x A100

Dialogue scripts can be generated with `openai/gpt-oss-120b` behind a local
vLLM OpenAI-compatible server. This path is opt-in through
`GEMMA_BACKEND=openai-compatible`; the existing Gemma
`transformers-subprocess` backend remains the default fallback path.

The redesigned prompt embeds all five curated examples from
`tests/fixtures/listening_dialogues.jsonl` and explicitly trains for active
listening, emotion mirroring, varied Japanese backchannels, reflective
paraphrase/summary, and gentle open-ended probing. It also varies openings and
response structure while prioritizing concrete `content_seeds`.

Submit the two-A100 dialogue-only job:

```bash
qsub -V scripts/run_dialogues_gptoss_2a100.pbs
```

The PBS script starts vLLM with tensor parallelism 2 in an isolated uv
environment, waits up to 10 minutes for `/v1/models`, then runs
`use_cases,dialogues,enrich` without audio. Output is written under
`data/runs/<BATCH_ID>/shard_<index>/`.

Before submission, verify that `#PBS -l select=1:res=middle` allocates exactly
two A100 GPUs on the target cluster. `PORT`, `MAX_MODEL_LEN`, `VLLM_MODEL`,
`NUM_CASES`, `BASE_SEED`, `BATCH_ID`, and `OUT_ROOT` are configurable.

## Generate a dataset

On the GPU server:

```bash
NUM_CASES=140 \
RUN_ID=full_duplex_v1 \
bash scripts/run_full_duplex_training_data.sh
```

PBS:

```bash
qsub scripts/run_full_duplex_training_data.pbs

# Small run
qsub -v NUM_CASES=14 scripts/run_full_duplex_training_data.pbs
```

The PBS job uses one V100 and defaults Gemma/Qwen3-TTS to `float16`.
Its output is written under `data/runs/fd_train_<timestamp>[_jobid]/`.

Generate only use cases and Gemma dialogue scripts:

```bash
NUM_CASES=140 \
RUN_ID=full_duplex_v1 \
STEPS=use_cases,dialogues \
bash scripts/run_full_duplex_training_data.sh
```

Template fallback is disabled by default because repeated fallback dialogues
reduce training diversity. The deterministic template backend can be used for
a dialogue-schema smoke test:

```bash
GEMMA_BACKEND=template \
NUM_CASES=14 \
RUN_ID=full_duplex_smoke \
STEPS=use_cases,dialogues \
bash scripts/run_full_duplex_training_data.sh
```

Outputs:

```text
data/runs/<RUN_ID>/
├── use_cases.jsonl
├── gemma_dialogues/
│   └── dialogues.jsonl
└── training_set/
    ├── synthetic_moshi_train.jsonl
    ├── dialogues.jsonl
    └── data_stereo/
        ├── sample_*.wav
        └── sample_*.json
```

The run can be passed to the existing fine-tuning launchers:

```bash
SRC_RUN_DIR=data/runs/full_duplex_v1 qsub scripts/fullft_sweep.pbs
```

## Timing schema

Normal turns remain sequential. A turn can additionally specify:

- `timing: "overlap_previous"`: start during the immediately preceding audio
  turn.
- `start_after_previous_start_sec`: overlap start offset.
- `truncate_previous_after_sec`: stop the preceding audio shortly after an
  interruption begins.
- `gain`: amplitude multiplier, mainly for background speech.
- `voice_role`: `user`, `other`, or `background`.
- `event`: behavior label such as `user_interruption`.

`generate_qwen3_tts_data.py` validates every dialogue carrying a
`duplex_task`. Invalid or missing timing events fail before the Qwen3-TTS model
is loaded.

The resulting WAVs contain actual simultaneous left/right channel audio for
backchannels, interruptions, side speech, and background speech. This differs
from the fixed evaluation adapter, whose current inputs are mostly arranged
serially on a mono timeline.
