# Japanese full-duplex training data

The 14 fixed cases in `eval_sets/full_duplex_ja/scenarios.jsonl` are evaluation
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
   reflects natural Japanese turn-taking.
6. The usual Moshi fine-tuning launchers consume
   `training_set/synthetic_moshi_train.jsonl`.

## Large-scale (~1000h) generation

`run_full_duplex_training_data_1000h.pbs` is a PBS job array. Each sub-job is a
self-contained shard (use_cases -> dialogues -> enrich -> audio) on one V100,
writing to its own directory. With eight V100s the scheduler runs eight shards
at a time and queues the rest, so no special throttling is needed. Self-
contained shards are resumable: just re-submit failed indices.

Minimal config to reach ~1000h within ~2 days on 8x V100:

- 96 shards x 250 dialogues = 24,000 dialogues (~1000h at ~150s average).
- Required aggregate throughput is ~2.6x real time (Gemma + Qwen3-TTS). Eight
  V100s fully used is the minimum that meets the deadline; fewer cards will not.
  If real RTF is slower, extend `walltime`, lower `DIALOGUES_PER_SHARD`, or
  accept ~600-700h.

```bash
# Always run one pilot shard first to measure real throughput and the actual
# average dialogue length before committing all 96 shards.
qsub -J 0-0 scripts/run_full_duplex_training_data_1000h.pbs

# Full run.
qsub scripts/run_full_duplex_training_data_1000h.pbs

# After all shards finish, merge into one manifest (CPU only, login node):
uv run python scripts/merge_training_shards.py \
  --batch-dir data/runs/fd_1000h \
  --out-dir   data/runs/fd_1000h/merged

# Fine-tune with the merged dataset:
SRC_RUN_DIR=data/runs/fd_1000h/merged qsub scripts/fullft_sweep.pbs
```

The merge writes absolute wav paths, so no audio is copied. Useful overrides:
`BATCH_ID`, `DIALOGUES_PER_SHARD`, `BASE_SEED`, `LISTENING_RATIO`, `GAP_SEC`.

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
Its output is written under `data/runs/fd_train_<PBS_JOBID>/`.

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
